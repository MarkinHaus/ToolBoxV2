#!/usr/bin/env bash

# ============================================================
# ToolBoxV2 Installer v1.0.0
# https://github.com/MarkinHaus/ToolBoxV2
#
# Usage:
#   curl -fsSL https://get.simplecore.app | bash
#   bash installer.sh [--config tb-install.yaml] [--mode native|uv|docker|source]
#   bash installer.sh --uninstall
#   bash installer.sh --update
# ============================================================

echo "**************************************************************************"
echo "***████████╗*██████╗***██████╗**██╗*********██████╗***██████╗*██╗***██╗***"
echo "***╚══██╔══╝██╔═══██╗*██╔═══██╗*██║*********██╔══██╗*██╔═══██╗*╚██╗██╔╝***"
echo "******██║***██║***██║*██║***██║*██║*********██████╔╝*██║***██║**╚███╔╝****"
echo "******██║***██║***██║*██║***██║*██║*********██╔══██╗*██║***██║**██╔██╗****"
echo "******██║***╚██████╔╝*╚██████╔╝*███████╗****██████╔╝*╚██████╔╝*██╔╝*██╗***"
echo "******╚═╝****╚═════╝***╚═════╝**╚══════╝****╚═════╝***╚═════╝**╚═╝**╚═╝***"
echo "**************************************************************************"
echo "Zero the Hero - ToolBoxV2 Core Installer"

set -euo pipefail

# ── Constants ────────────────────────────────────────────────
INSTALLER_VERSION="1.0.0"
TB_ARTIFACT_NAME="ToolBoxV2"
REGISTRY_API="https://registry.simplecore.app/api/v1"
GITHUB_REPO="MarkinHaus/ToolBoxV2"
GITHUB_API="https://api.github.com/repos/${GITHUB_REPO}"
GITHUB_RAW="https://raw.githubusercontent.com/${GITHUB_REPO}/refs/heads/master"
MIN_DISK_MB=300
FEATURES_IMMUTABLE="mini core"
FEATURES_OPTIONAL="cli web desktop isaa exotic"

# ── Colors ───────────────────────────────────────────────────
if [ -t 1 ]; then
  R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m'
  B='\033[0;34m' C='\033[0;36m' D='\033[2m' BOLD='\033[1m' NC='\033[0m'
else
  R='' G='' Y='' B='' C='' D='' BOLD='' NC=''
fi

log()   { echo -e "${G}[✓]${NC} $*"; }
info()  { echo -e "${B}[→]${NC} $*"; }
warn()  { echo -e "${Y}[!]${NC} $*"; }
fail()  { echo -e "${R}[✗]${NC} $*" >&2; exit 1; }
ask()   { echo -e "${C}[?]${NC} $*"; }
step()  { echo -e "\n${BOLD}${B}── $* ${NC}${D}──────────────────────────────────────${NC}"; }

# ── Global State ─────────────────────────────────────────────
OS=""           # linux | macos | windows
ARCH=""         # x86_64 | arm64
INSTALL_MODE="" # native | uv | docker | source
SOURCE_FROM=""  # git | registry
SOURCE_BRANCH="main"
INSTALL_PATH=""
ENVIRONMENT="development"
INSTANCE_ID="tbv2_main"
FEATURES="core cli"
OPT_NGINX=false
OPT_DOCKER=false
OPT_OLLAMA=false
OPT_MINIO=false
OPT_REGISTRY=false
REGISTRY_URL="${REGISTRY_API%/api/v1}"
RUNTIME=""      # uv | venv | none
UV_BIN=""
PYTHON_BIN=""
REGISTRY_REACHABLE=false
ACTION="install" # install | update | uninstall

# ── Args ─────────────────────────────────────────────────────
CONFIG_FILE=""
for arg in "$@"; do
  case $arg in
    --uninstall)           ACTION="uninstall" ;;
    --update)              ACTION="update" ;;
    --config=*)            CONFIG_FILE="${arg#*=}" ;;
    --config)              shift; CONFIG_FILE="$1" ;;
    --mode=*)              INSTALL_MODE="${arg#*=}" ;;
    --mode)                shift; INSTALL_MODE="$1" ;;
    --path=*)              INSTALL_PATH="${arg#*=}" ;;
    --help|-h)
      echo "Usage: bash installer.sh [options]"
      echo "  --config <file>   Load install config from YAML"
      echo "  --mode <mode>     native | uv | docker | source"
      echo "  --path <dir>      Custom install directory"
      echo "  --update          Update existing installation"
      echo "  --uninstall       Remove ToolBoxV2"
      exit 0 ;;
  esac
done

# ============================================================
# HELPERS
# ============================================================

os_default_path() {
  case "$OS" in
    linux)  echo "/opt/toolboxv2" ;;
    macos)  echo "${HOME}/Library/Application Support/toolboxv2" ;;
    *)      echo "${HOME}/.toolboxv2" ;;
  esac
}

appdata_path() {
  case "$OS" in
    linux)  echo "${HOME}/.local/share/toolboxv2" ;;
    macos)  echo "${HOME}/Library/Application Support/toolboxv2" ;;
    *)      echo "${HOME}/.toolboxv2" ;;
  esac
}

bin_dir() {
  case "$OS" in
    linux|macos) echo "/usr/local/bin" ;;
    *)           echo "${HOME}/.local/bin" ;;
  esac
}

# Simple YAML field reader (no yq dependency)
yaml_get() {
  local file="$1" key="$2" default="${3:-}"
  local val
  val=$(grep -E "^${key}:" "$file" 2>/dev/null | head -1 | sed 's/^[^:]*: *//' | tr -d '"' || true)
  echo "${val:-$default}"
}

confirm() {
  local msg="$1" default="${2:-y}"
  local prompt
  [ "$default" = "y" ] && prompt="[Y/n]" || prompt="[y/N]"
  ask "${msg} ${prompt}: "
  read -r reply
  reply="${reply:-$default}"
  [[ "$reply" =~ ^[Yy] ]]
}

prompt_with_default() {
  local msg="$1" default="$2"
  ask "${msg} (default: ${C}${default}${NC}): "
  read -r reply
  echo "${reply:-$default}"
}

check_disk_space() {
  local path="$1"
  local parent="$path"
  while [ ! -d "$parent" ]; do parent="$(dirname "$parent")"; done
  local avail
  avail=$(df -m "$parent" 2>/dev/null | awk 'NR==2{print $4}' || echo 9999)
  [ "$avail" -ge "$MIN_DISK_MB" ] || fail "Not enough disk space (need ${MIN_DISK_MB}MB, have ${avail}MB at ${parent})"
}

download() {
  local url="$1" dest="$2"
  if command -v curl &>/dev/null; then
    curl -fsSL --progress-bar "$url" -o "$dest"
  elif command -v wget &>/dev/null; then
    wget -q --show-progress "$url" -O "$dest"
  else
    fail "Neither curl nor wget found"
  fi
}

registry_get() {
  local path="$1"
  if command -v curl &>/dev/null; then
    curl -fsSL "${REGISTRY_API}${path}" 2>/dev/null || true
  else
    wget -qO- "${REGISTRY_API}${path}" 2>/dev/null || true
  fi
}

# ============================================================
# PHASE 0 — DISCOVERY
# ============================================================
phase_discovery() {
  step "Phase 0 — Discovery"

  # OS + Arch
  case "$(uname -s)" in
    Linux*)  OS="linux" ;;
    Darwin*) OS="macos" ;;
    MINGW*|CYGWIN*|MSYS*) OS="windows" ;;
    *)       fail "Unsupported OS: $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) ARCH="x86_64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *) fail "Unsupported architecture: $(uname -m)" ;;
  esac
  info "Platform: ${OS}/${ARCH}"

  # Scan for existing installations
  local found_installs=()
  local scan_paths=(
    "/opt/toolboxv2"
    "${HOME}/.local/share/toolboxv2"
    "${HOME}/.toolboxv2"
    "${HOME}/Library/Application Support/toolboxv2"
    "${TOOLBOX_HOME:-__unset__}"
    "${TB_INSTALL_DIR:-__unset__}"
  )

  for p in "${scan_paths[@]}"; do
    [ "$p" = "__unset__" ] && continue
    if [ -f "${p}/install.manifest" ]; then
      found_installs+=("$p")
    fi
  done

  # Check PATH for existing tb binary
  if command -v tb &>/dev/null; then
    local tb_path
    tb_path=$(command -v tb)
    info "Existing 'tb' binary found: ${tb_path}"
    found_installs+=("$(dirname "$(dirname "$tb_path")")")
  fi

  # Check Docker for TB images
  if command -v docker &>/dev/null; then
    local docker_images
    docker_images=$(docker images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | grep -i toolboxv2 || true)
    [ -n "$docker_images" ] && info "Docker TB image(s) found: ${docker_images}"
  fi

  if [ ${#found_installs[@]} -gt 0 ]; then
    echo ""
    warn "Existing ToolBoxV2 installation(s) found:"
    for p in "${found_installs[@]}"; do
      local ver
      ver=$(yaml_get "${p}/install.manifest" "tb_version" "unknown")
      local mode
      mode=$(yaml_get "${p}/install.manifest" "install_mode" "unknown")
      echo "  ${C}${p}${NC}  (${mode}, v${ver})"
    done
    echo ""
    if [ "$ACTION" = "install" ]; then
      if confirm "Update existing installation?"; then
        ACTION="update"
        INSTALL_PATH="${found_installs[0]}"
      else
        confirm "Install fresh alongside?" || fail "Aborted."
      fi
    fi
  else
    info "No existing installation found — fresh install"
  fi
}

# ============================================================
# PHASE 1 — CONFIG
# ============================================================
phase_config() {
  step "Phase 1 — Configuration"

  # Load config file if given or found
  if [ -z "$CONFIG_FILE" ] && [ -f "tb-install.yaml" ]; then
    CONFIG_FILE="tb-install.yaml"
    info "Found tb-install.yaml in current directory"
  fi

  if [ -n "$CONFIG_FILE" ]; then
    [ -f "$CONFIG_FILE" ] || fail "Config file not found: ${CONFIG_FILE}"
    info "Loading config: ${CONFIG_FILE}"
    [ -z "$INSTALL_MODE" ]  && INSTALL_MODE=$(yaml_get "$CONFIG_FILE" "install_mode" "")
    [ -z "$INSTALL_PATH" ]  && INSTALL_PATH=$(yaml_get "$CONFIG_FILE" "install_path" "")
    SOURCE_FROM=$(yaml_get "$CONFIG_FILE" "source_from" "git")
    SOURCE_BRANCH=$(yaml_get "$CONFIG_FILE" "source_branch" "main")
    ENVIRONMENT=$(yaml_get "$CONFIG_FILE" "environment" "development")
    INSTANCE_ID=$(yaml_get "$CONFIG_FILE" "instance_id" "tbv2_main")
    OPT_NGINX=$(yaml_get "$CONFIG_FILE" "optional.nginx" "false")
    OPT_DOCKER=$(yaml_get "$CONFIG_FILE" "optional.docker_runtime" "false")
    OPT_OLLAMA=$(yaml_get "$CONFIG_FILE" "optional.ollama" "false")
    OPT_MINIO=$(yaml_get "$CONFIG_FILE" "optional.minio" "false")
    OPT_REGISTRY=$(yaml_get "$CONFIG_FILE" "optional.registry" "false")

    # Features from config (space-separated line under features:)
    local feat_line
    feat_line=$(grep -A1 "^features:" "$CONFIG_FILE" 2>/dev/null | tail -1 | tr -d ' []' | tr ',' ' ' || true)
    [ -n "$feat_line" ] && FEATURES="$feat_line"
  fi

  # Interactive fallback for each missing value
  # Install mode
  if [ -z "$INSTALL_MODE" ]; then
    echo ""
    echo "  Select install mode:"
    echo "  ${C}1)${NC} native   — Single binary, no Python required ${D}(recommended)${NC}"
    echo "  ${C}2)${NC} uv       — Python package via uv tool"
    echo "  ${C}3)${NC} docker   — Containerized, isolated"
    echo "  ${C}4)${NC} source   — Full source from Git or Registry"
    ask "Mode [1-4] (default: 1): "
    read -r mode_choice
    case "${mode_choice:-1}" in
      1) INSTALL_MODE="native" ;;
      2) INSTALL_MODE="uv" ;;
      3) INSTALL_MODE="docker" ;;
      4) INSTALL_MODE="source" ;;
      *) INSTALL_MODE="native" ;;
    esac
  fi

  # source_from sub-selection
  if [ "$INSTALL_MODE" = "source" ] && [ -z "$SOURCE_FROM" ]; then
    echo "  Source from:"
    echo "  ${C}1)${NC} git      — Clone from GitHub (editable dev tree)"
    echo "  ${C}2)${NC} registry — Download release tarball"
    ask "Source [1-2] (default: 1): "
    read -r src_choice
    [ "${src_choice:-1}" = "2" ] && SOURCE_FROM="registry" || SOURCE_FROM="git"
  fi
  SOURCE_FROM="${SOURCE_FROM:-git}"

  # Features
  echo ""
  echo "  Included (always): ${G}mini core${NC}"
  echo "  Optional features:"
  local sel_features="$FEATURES"
  for feat in $FEATURES_OPTIONAL; do
    local already=false
    echo "$sel_features" | grep -qw "$feat" && already=true
    local currently
    $already && currently="${G}yes${NC}" || currently="${D}no${NC}"
    if confirm "  Enable ${BOLD}${feat}${NC}? [currently: ${currently}]" "$($already && echo y || echo n)"; then
      echo "$sel_features" | grep -qw "$feat" || sel_features="$sel_features $feat"
    else
      sel_features=$(echo "$sel_features" | tr ' ' '\n' | grep -v "^${feat}$" | tr '\n' ' ')
    fi
  done
  FEATURES="$sel_features"

  # Environment
  if [ -z "$ENVIRONMENT" ] || [ "$ENVIRONMENT" = "development" ]; then
    ENVIRONMENT=$(prompt_with_default "Environment" "development")
  fi

  # Install path
  local default_path
  default_path=$(os_default_path)
  if [ -z "$INSTALL_PATH" ]; then
    ask "Custom install path? Leave empty for default (${C}${default_path}${NC}): "
    read -r custom_path
    INSTALL_PATH="${custom_path:-$default_path}"
  fi
  INSTALL_PATH="${INSTALL_PATH:-$default_path}"

  log "Mode: ${INSTALL_MODE}${INSTALL_MODE:+ }${SOURCE_FROM:+(${SOURCE_FROM})}"
  log "Path: ${INSTALL_PATH}"
  log "Env:  ${ENVIRONMENT}"
  log "Features: ${FEATURES_IMMUTABLE} ${FEATURES}"
}

# ============================================================
# PHASE 2 — PRE-FLIGHT
# ============================================================
phase_preflight() {
  step "Phase 2 — Pre-flight Checks"

  # 1. Custom path validation
  info "Checking install path: ${INSTALL_PATH}"
  if [ -e "$INSTALL_PATH" ] && [ ! -d "$INSTALL_PATH" ]; then
    fail "${INSTALL_PATH} exists and is not a directory"
  fi
  check_disk_space "$INSTALL_PATH"
  # Write test
  mkdir -p "$INSTALL_PATH" 2>/dev/null || {
    warn "Cannot create ${INSTALL_PATH} — try with sudo or choose a different path"
    INSTALL_PATH=$(prompt_with_default "Enter writable install path" "$(appdata_path)")
    mkdir -p "$INSTALL_PATH" || fail "Cannot create ${INSTALL_PATH}"
  }
  local testfile="${INSTALL_PATH}/.tb_write_test"
  touch "$testfile" 2>/dev/null || {
    INSTALL_PATH=$(prompt_with_default "Path not writable. Enter writable path" "$(appdata_path)")
    mkdir -p "$INSTALL_PATH" && touch "${INSTALL_PATH}/.tb_write_test" || fail "Cannot write to ${INSTALL_PATH}"
    testfile="${INSTALL_PATH}/.tb_write_test"
  }
  rm -f "$testfile"
  log "Install path OK: ${INSTALL_PATH}"

  # 2. DNS / network
  info "Checking network..."
  if command -v curl &>/dev/null; then
    curl -fsS --max-time 3 "https://1.1.1.1" -o /dev/null 2>/dev/null || \
    curl -fsS --max-time 3 "https://github.com" -o /dev/null 2>/dev/null || \
      fail "No network connectivity. Installer requires internet access."
  else
    wget -q --timeout=3 -O /dev/null "https://1.1.1.1" 2>/dev/null || \
      fail "No network connectivity."
  fi
  log "Network OK"

  # 3. Registry reachable?
  info "Checking registry..."
  local health
  health=$(registry_get "/health" || true)
  if echo "$health" | grep -q '"healthy"'; then
    REGISTRY_REACHABLE=true
    log "Registry OK (${REGISTRY_API})"
  else
    warn "Registry unreachable — using GitHub Releases as fallback"
    REGISTRY_REACHABLE=false
  fi

  # 4. Runtime detection (native/docker skip this)
  if [ "$INSTALL_MODE" = "native" ] || [ "$INSTALL_MODE" = "docker" ]; then
    RUNTIME="none"
    log "Runtime: none required for ${INSTALL_MODE} mode"
  else
    # Try uv first
    if command -v uv &>/dev/null; then
      UV_BIN=$(command -v uv)
      RUNTIME="uv"
      log "Runtime: uv found at ${UV_BIN}"
    else
      # Try Python 3.11+
      for py in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$py" &>/dev/null; then
          local pyver
          pyver=$("$py" -c "import sys; print(sys.version_info >= (3,11))" 2>/dev/null || echo "False")
          if [ "$pyver" = "True" ]; then
            PYTHON_BIN=$(command -v "$py")
            RUNTIME="venv"
            log "Runtime: Python found at ${PYTHON_BIN} (venv fallback)"
            break
          fi
        fi
      done

      # Bootstrap uv if still no runtime
      if [ -z "$RUNTIME" ]; then
        info "No runtime found — bootstrapping uv..."
        if command -v curl &>/dev/null; then
          curl -LsSf https://astral.sh/uv/install.sh | sh
        else
          wget -qO- https://astral.sh/uv/install.sh | sh
        fi
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:$PATH"
        command -v uv &>/dev/null || fail "uv bootstrap failed — install manually: https://docs.astral.sh/uv/"
        UV_BIN=$(command -v uv)
        RUNTIME="uv"
        log "Runtime: uv bootstrapped at ${UV_BIN}"
      fi
    fi
  fi

  # 5. Optional tools (config → fragen wenn nicht gesetzt)
  _check_optional "docker"  "$OPT_DOCKER"  "Docker runtime"  "docker --version"
  _check_optional "nginx"   "$OPT_NGINX"   "nginx"           "nginx -v"
  _check_optional "ollama"  "$OPT_OLLAMA"  "Ollama (LLM)"    "ollama --version"
}

_check_optional() {
  local name="$1" cfg_val="$2" label="$3" test_cmd="$4"
  local present=false
  command -v "$name" &>/dev/null && present=true

  case "$cfg_val" in
    true)
      $present && { log "${label} already installed"; return; }
      info "Installing ${label} (required by config)..."
      _install_optional "$name"
      ;;
    false)
      $present && info "${label} present (skipped by config)" || true
      ;;
    "")
      if ! $present; then
        if confirm "${label} not found. Install it?"; then
          _install_optional "$name"
        fi
      else
        log "${label} found"
      fi
      ;;
  esac
}

_install_optional() {
  local name="$1"
  case "$name" in
    docker)
      case "$OS" in
        linux)  curl -fsSL https://get.docker.com | sh ;;
        macos)  warn "Install Docker Desktop from https://docker.com/products/docker-desktop" ;;
      esac ;;
    nginx)
      case "$OS" in
        linux)
          if command -v apt-get &>/dev/null; then sudo apt-get install -y nginx
          elif command -v dnf &>/dev/null; then sudo dnf install -y nginx; fi ;;
        macos) brew install nginx ;;
      esac ;;
    ollama)
      curl -fsSL https://ollama.ai/install.sh | sh ;;
    *)
      warn "Don't know how to install ${name} — please install manually" ;;
  esac
}

# ============================================================
# PHASE 3 — INSTALL
# ============================================================
phase_install() {
  step "Phase 3 — Install (${INSTALL_MODE})"

  mkdir -p "${INSTALL_PATH}/bin" "${INSTALL_PATH}/.data" \
           "${INSTALL_PATH}/.config" "${INSTALL_PATH}/logs"

  case "$INSTALL_MODE" in
    native)   _install_native ;;
    uv)       _install_uv ;;
    docker)   _install_docker ;;
    source)   _install_source ;;
  esac
}

_get_version_and_url() {
  # Returns: "VERSION URL CHECKSUM" — Registry first, GitHub fallback
  local platform="$OS"
  local arch_tag
  [ "$ARCH" = "x86_64" ] && arch_tag="x64" || arch_tag="arm64"

  if $REGISTRY_REACHABLE; then
    local resp
    resp=$(registry_get "/artifacts/${TB_ARTIFACT_NAME}/latest?platform=${platform}&architecture=${ARCH}" || true)
    if [ -n "$resp" ]; then
      local version url checksum
      version=$(echo "$resp" | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)
      checksum=$(echo "$resp" | grep -o '"checksum":"[^"]*"' | head -1 | cut -d'"' -f4)
      # Get signed download URL
      local dl_resp
      dl_resp=$(registry_get "/artifacts/${TB_ARTIFACT_NAME}/latest?platform=${platform}&architecture=${ARCH}" || true)
      url=$(echo "$dl_resp" | grep -o '"url":"[^"]*"' | head -1 | cut -d'"' -f4)
      [ -n "$url" ] && { echo "$version $url $checksum"; return; }
    fi
  fi

  # GitHub Releases fallback
  info "Using GitHub Releases..."
  local tag
  tag=$(download "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" /tmp/tb_release.json 2>/dev/null; \
        grep -o '"tag_name":"[^"]*"' /tmp/tb_release.json | head -1 | cut -d'"' -f4)
  local fname="toolbox-${platform}-${arch_tag}"
  [ "$OS" = "windows" ] && fname="${fname}.exe"
  local url="https://github.com/${GITHUB_REPO}/releases/download/${tag}/${fname}"
  echo "${tag#v} $url "
}

_install_native() {
  info "Fetching latest binary..."
  read -r tb_version dl_url checksum <<< "$(_get_version_and_url)"
  [ -z "$dl_url" ] && fail "Could not determine download URL"
  info "Downloading ToolBoxV2 v${tb_version}..."

  local bin_dest="${INSTALL_PATH}/bin/tb"
  download "$dl_url" "$bin_dest"
  chmod +x "$bin_dest"

  # Verify checksum if available
  if [ -n "$checksum" ] && command -v sha256sum &>/dev/null; then
    local actual
    actual=$(sha256sum "$bin_dest" | awk '{print $1}')
    [ "$actual" = "$checksum" ] || fail "Checksum mismatch — download may be corrupted"
    log "Checksum verified"
  fi

  # Symlink to system bin
  local sys_bin
  sys_bin=$(bin_dir)
  mkdir -p "$sys_bin"
  ln -sf "$bin_dest" "${sys_bin}/tb" 2>/dev/null || \
    warn "Could not symlink to ${sys_bin}/tb — add ${INSTALL_PATH}/bin to PATH manually"

  log "Binary installed: ${bin_dest}"
  echo "$tb_version"
}

_install_uv() {
  info "Installing via uv tool..."
  local pkg="ToolBoxV2"
  # Build extras string from features
  local extras
  extras=$(echo "$FEATURES" | tr ' ' '\n' | grep -v -E '^(core|mini)$' | paste -sd',' -)
  [ -n "$extras" ] && pkg="ToolBoxV2[${extras}]"

  "$UV_BIN" tool install "$pkg" --force
  log "Installed: ${pkg}"

  # Ensure uv tool bin is in PATH
  local uv_bin_dir
  uv_bin_dir=$("$UV_BIN" tool dir 2>/dev/null || echo "${HOME}/.local/bin")
  export PATH="${uv_bin_dir}:$PATH"

  local tb_ver
  tb_ver=$(tb --version 2>/dev/null | grep -o '[0-9][0-9.]*' | head -1 || echo "unknown")
  echo "$tb_ver"
}

_install_venv() {
  # pip/venv fallback path
  info "Installing via pip/venv..."
  local venv_path="${INSTALL_PATH}/.venv"
  "$PYTHON_BIN" -m venv "$venv_path"
  local pip="${venv_path}/bin/pip"
  "$pip" install --upgrade pip -q

  local pkg="ToolBoxV2"
  local extras
  extras=$(echo "$FEATURES" | tr ' ' '\n' | grep -v -E '^(core|mini)$' | paste -sd',' -)
  [ -n "$extras" ] && pkg="ToolBoxV2[${extras}]"

  "$pip" install "$pkg" -q
  log "Installed: ${pkg} in ${venv_path}"

  # Write wrapper script
  cat > "${INSTALL_PATH}/bin/tb" << EOF
#!/usr/bin/env bash
exec "${venv_path}/bin/tb" "\$@"
EOF
  chmod +x "${INSTALL_PATH}/bin/tb"

  local tb_ver
  tb_ver=$("${venv_path}/bin/tb" --version 2>/dev/null | grep -o '[0-9][0-9.]*' | head -1 || echo "unknown")
  echo "$tb_ver"
}

_install_docker() {
  command -v docker &>/dev/null || fail "Docker not found — run pre-flight or install Docker first"
  local image="ghcr.io/markinhaus/toolboxv2:latest"
  info "Pulling Docker image: ${image}"
  docker pull "$image"

  # Write tb wrapper script
  cat > "${INSTALL_PATH}/bin/tb" << EOF
#!/usr/bin/env bash
exec docker run --rm -it \\
  -v "${INSTALL_PATH}/.data:/data" \\
  -e TB_DATA_DIR=/data \\
  --name toolboxv2 \\
  ${image} "\$@"
EOF
  chmod +x "${INSTALL_PATH}/bin/tb"

  local sys_bin
  sys_bin=$(bin_dir)
  ln -sf "${INSTALL_PATH}/bin/tb" "${sys_bin}/tb" 2>/dev/null || true
  log "Docker wrapper installed"
  echo "latest"
}

_install_source() {
  info "Installing from source (${SOURCE_FROM})..."
  local src_dir="${INSTALL_PATH}/src"
  local tb_ver="unknown"

  if [ "$SOURCE_FROM" = "git" ]; then
    command -v git &>/dev/null || fail "git not found — install git first"
    if [ -d "${src_dir}/.git" ]; then
      info "Existing repo found — pulling..."
      git -C "$src_dir" fetch origin
      git -C "$src_dir" checkout "$SOURCE_BRANCH"
      git -C "$src_dir" pull origin "$SOURCE_BRANCH"
    else
      git clone --branch "$SOURCE_BRANCH" --depth 1 \
        "https://github.com/${GITHUB_REPO}.git" "$src_dir"
    fi
  else
    # Registry tarball
    [ -z "$1" ] && tb_ver="latest" || tb_ver="$1"
    local resp dl_url
    resp=$(registry_get "/packages/ToolBoxV2/versions/${tb_ver}/download" || true)
    dl_url=$(echo "$resp" | grep -o '"url":"[^"]*"' | head -1 | cut -d'"' -f4)
    [ -z "$dl_url" ] && fail "Could not get tarball URL from registry"
    info "Downloading source tarball..."
    download "$dl_url" /tmp/tbv2_src.tar.gz
    mkdir -p "$src_dir"
    tar -xzf /tmp/tbv2_src.tar.gz -C "$src_dir" --strip-components=1
    rm /tmp/tbv2_src.tar.gz
  fi

  # Sync deps
  cd "$src_dir"
  if [ "$RUNTIME" = "uv" ]; then
    local extras
    extras=$(echo "$FEATURES" | tr ' ' '\n' | grep -v -E '^(core|mini)$' | paste -sd',' -)
    if [ -n "$extras" ]; then
      "$UV_BIN" sync --extra "$extras"
    else
      "$UV_BIN" sync
    fi
    # Write wrapper
    cat > "${INSTALL_PATH}/bin/tb" << EOF
#!/usr/bin/env bash
cd "${src_dir}" && exec "${UV_BIN}" run tb "\$@"
EOF
  else
    "$PYTHON_BIN" -m venv "${src_dir}/.venv"
    "${src_dir}/.venv/bin/pip" install -e ".[${FEATURES// /,}]" -q
    cat > "${INSTALL_PATH}/bin/tb" << EOF
#!/usr/bin/env bash
exec "${src_dir}/.venv/bin/tb" "\$@"
EOF
  fi
  chmod +x "${INSTALL_PATH}/bin/tb"
  ln -sf "${INSTALL_PATH}/bin/tb" "$(bin_dir)/tb" 2>/dev/null || true

  tb_ver=$(${INSTALL_PATH}/bin/tb --version 2>/dev/null | grep -o '[0-9][0-9.]*' | head -1 || echo "dev")
  cd - >/dev/null
  echo "$tb_ver"
}

# ============================================================
# PHASE 4 — MANIFEST & ENV
# ============================================================
phase_write_manifests() {
  local tb_version="$1"
  step "Phase 4 — Writing Manifests & Env"

  # install.manifest
  cat > "${INSTALL_PATH}/install.manifest" << EOF
installer_version: "${INSTALLER_VERSION}"
installed_at: "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
install_mode: "${INSTALL_MODE}"
source_from: "${SOURCE_FROM}"
source_branch: "${SOURCE_BRANCH}"
tb_version: "${tb_version}"
toolbox_home: "${INSTALL_PATH}"
bin_path: "${INSTALL_PATH}/bin/tb"
src_path: "${INSTALL_PATH}/src"
venv_path: "${INSTALL_PATH}/.venv"
runtime: "${RUNTIME}"
python_path: "${PYTHON_BIN}"
uv_path: "${UV_BIN}"
features: "${FEATURES_IMMUTABLE} ${FEATURES}"
optional_installed: "$(
  opts=""
  $OPT_NGINX && opts="$opts nginx"
  $OPT_DOCKER && opts="$opts docker"
  $OPT_OLLAMA && opts="$opts ollama"
  $OPT_MINIO && opts="$opts minio"
  $OPT_REGISTRY && opts="$opts registry"
  echo $opts
)"
EOF
  log "Written: ${INSTALL_PATH}/install.manifest"

  # tb-manifest.yaml (only write if not already present)
  local manifest_file="${INSTALL_PATH}/tb-manifest.yaml"
  if [ ! -f "$manifest_file" ]; then
    download "${GITHUB_RAW}/toolboxv2/utils/manifest/example-manifest.yaml" \
             "${manifest_file}" 2>/dev/null || \
    cat > "$manifest_file" << EOF
manifest_version: "1.0.0"
app:
  name: ToolBoxV2
  version: "${tb_version}"
  instance_id: "${INSTANCE_ID}"
  environment: ${ENVIRONMENT}
  debug: false
  log_level: INFO
paths:
  data_dir: "${INSTALL_PATH}/.data"
  config_dir: "${INSTALL_PATH}/.config"
  logs_dir: "${INSTALL_PATH}/logs"
  mods_dir: "${INSTALL_PATH}/mods"
  dist_dir: "${INSTALL_PATH}/dist"
registry:
  url: "${REGISTRY_URL}"
  auto_update: false
EOF
    log "Written: ${manifest_file}"
  else
    info "tb-manifest.yaml already exists — skipping (run 'tb manifest apply' to update)"
  fi

  # .env (only missing vars)
  local env_file="${INSTALL_PATH}/.env"
  if [ ! -f "$env_file" ]; then
    cat > "$env_file" << EOF
# ToolBoxV2 Environment — generated by installer
TOOLBOX_HOME="${INSTALL_PATH}"
TB_INSTALL_DIR="${INSTALL_PATH}"
TB_DATA_DIR="${INSTALL_PATH}/.data"
TB_DIST_DIR="${INSTALL_PATH}/dist"
TB_ENV="${ENVIRONMENT}"
TB_JWT_SECRET=                    # REQUIRED: set before production use
TB_COOKIE_SECRET=                 # REQUIRED: set before production use
TOOLBOXV2_BASE=localhost
TOOLBOXV2_BASE_PORT=8000
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
EOF
    log "Written: ${env_file}"
    warn "Edit ${env_file} to set TB_JWT_SECRET and TB_COOKIE_SECRET before running in production"
  fi

  # Write TOOLBOX_HOME to shell profile
  local profile_file
  case "$OS" in
    linux|macos)
      profile_file="${HOME}/.bashrc"
      [ -f "${HOME}/.zshrc" ] && profile_file="${HOME}/.zshrc"
      ;;
  esac
  if [ -n "${profile_file:-}" ]; then
    if ! grep -q "TOOLBOX_HOME" "$profile_file" 2>/dev/null; then
      echo "" >> "$profile_file"
      echo "# ToolBoxV2" >> "$profile_file"
      echo "export TOOLBOX_HOME=\"${INSTALL_PATH}\"" >> "$profile_file"
      echo "export PATH=\"${INSTALL_PATH}/bin:\$PATH\"" >> "$profile_file"
      log "Added TOOLBOX_HOME to ${profile_file}"
    fi
  fi
}

# ============================================================
# UPDATE
# ============================================================
action_update() {
  step "Update"
  local manifest="${INSTALL_PATH}/install.manifest"
  [ -f "$manifest" ] || fail "No install.manifest found at ${INSTALL_PATH}"
  INSTALL_MODE=$(yaml_get "$manifest" "install_mode")
  RUNTIME=$(yaml_get "$manifest" "runtime")
  UV_BIN=$(yaml_get "$manifest" "uv_path")
  PYTHON_BIN=$(yaml_get "$manifest" "python_path")

  case "$INSTALL_MODE" in
    native)
      local new_ver
      new_ver=$(_install_native)
      phase_write_manifests "$new_ver"
      ;;
    uv)
      "$UV_BIN" tool upgrade ToolBoxV2
      log "Updated via uv"
      ;;
    docker)
      docker pull ghcr.io/markinhaus/toolboxv2:latest
      log "Updated Docker image"
      ;;
    source)
      SOURCE_FROM=$(yaml_get "$manifest" "source_from" "git")
      SOURCE_BRANCH=$(yaml_get "$manifest" "source_branch" "main")
      FEATURES=$(yaml_get "$manifest" "features" "core cli")
      _install_source
      log "Updated from source"
      ;;
  esac
}

# ============================================================
# UNINSTALL
# ============================================================
action_uninstall() {
  step "Uninstall"
  local manifest="${INSTALL_PATH}/install.manifest"
  [ -f "$manifest" ] || fail "No install.manifest found — cannot uninstall cleanly"

  local mode
  mode=$(yaml_get "$manifest" "install_mode")
  local home_path
  home_path=$(yaml_get "$manifest" "toolbox_home")

  warn "This will remove ToolBoxV2 from: ${home_path}"
  confirm "Continue with uninstall?" || fail "Aborted."

  case "$mode" in
    docker)
      docker rm -f toolboxv2 2>/dev/null || true
      docker rmi ghcr.io/markinhaus/toolboxv2:latest 2>/dev/null || true
      ;;
    uv)
      "$UV_BIN" tool uninstall ToolBoxV2 2>/dev/null || true
      ;;
  esac

  rm -f "$(bin_dir)/tb"
  confirm "Remove all data and config in ${home_path}? (${R}irreversible${NC})" "n" && rm -rf "$home_path"
  log "Uninstall complete"
}

# ============================================================
# SUMMARY
# ============================================================
print_summary() {
  local tb_version="$1"
  echo ""
  echo -e "${BOLD}${G}╔══════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${G}║   ToolBoxV2 v${tb_version} installed successfully   ║${NC}"
  echo -e "${BOLD}${G}╚══════════════════════════════════════════════════╝${NC}"
  echo ""
  echo -e "  ${BOLD}Home:${NC}     ${INSTALL_PATH}"
  echo -e "  ${BOLD}Mode:${NC}     ${INSTALL_MODE}"
  echo -e "  ${BOLD}Runtime:${NC}  ${RUNTIME}"
  echo -e "  ${BOLD}Features:${NC} ${FEATURES_IMMUTABLE} ${FEATURES}"
  echo ""
  echo -e "  ${C}Next steps:${NC}"
  echo -e "  1. Reload shell:  ${BOLD}source ~/.bashrc${NC} (or open new terminal)"
  echo -e "  2. First run:     ${BOLD}tb${NC}"
  echo -e "  3. Check status:  ${BOLD}tb status${NC}"
  if grep -q "TB_JWT_SECRET=$" "${INSTALL_PATH}/.env" 2>/dev/null; then
    echo ""
    echo -e "  ${Y}[!] Set secrets before production:${NC}"
    echo -e "      ${BOLD}${INSTALL_PATH}/.env${NC}"
  fi
  echo ""
}

# ============================================================
# MAIN
# ============================================================
main() {
  echo -e "${BOLD}${C}"
  echo "  ╔════════════════════════════════════════╗"
  echo "  ║     ToolBoxV2 Installer v${INSTALLER_VERSION}       ║"
  echo "  ╚════════════════════════════════════════╝"
  echo -e "${NC}"

  case "$ACTION" in
    install)
      phase_discovery
      phase_config
      phase_preflight

      local tb_version="unknown"
      # Route to correct install function
      if [ "$INSTALL_MODE" = "uv" ] && [ "$RUNTIME" = "venv" ]; then
        tb_version=$(_install_venv)
      else
        tb_version=$(phase_install)
      fi

      phase_write_manifests "$tb_version"
      print_summary "$tb_version"
      ;;
    update)
      phase_discovery
      action_update
      ;;
    uninstall)
      phase_discovery
      action_uninstall
      ;;
  esac
}

main "$@"
