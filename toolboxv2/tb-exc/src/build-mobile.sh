#!/bin/bash
# TB Language - Mobile Build Script
# Builds tb-runtime for Android and iOS platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PACKAGE="tb-runtime"
BUILD_TYPE="release"
FEATURES=""

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a target is installed
check_target() {
    local target=$1
    if rustup target list --installed | grep -q "^${target}$"; then
        return 0
    else
        return 1
    fi
}

# Function to install a target
install_target() {
    local target=$1
    print_info "Installing target: $target"
    rustup target add "$target"
}

# Function to build for a specific target
build_target() {
    local target=$1
    local platform=$2
    
    print_info "Building for $platform ($target)..."
    
    if [ -n "$FEATURES" ]; then
        cargo build --${BUILD_TYPE} --target "$target" -p "$PACKAGE" --features "$FEATURES"
    else
        cargo build --${BUILD_TYPE} --target "$target" -p "$PACKAGE"
    fi
    
    if [ $? -eq 0 ]; then
        print_info "✓ Successfully built for $platform"
    else
        print_error "✗ Failed to build for $platform"
        return 1
    fi
}

# Function to build all Android targets
build_android() {
    print_info "=== Building for Android ==="
    
    local android_targets=(
        "aarch64-linux-android:ARM64"
        "armv7-linux-androideabi:ARMv7"
        "i686-linux-android:x86"
        "x86_64-linux-android:x86_64"
    )
    
    for target_info in "${android_targets[@]}"; do
        IFS=':' read -r target name <<< "$target_info"
        
        if ! check_target "$target"; then
            print_warn "Target $target not installed"
            install_target "$target"
        fi
        
        build_target "$target" "Android $name"
    done
    
    print_info "=== Android build complete ==="
    print_info "Libraries located in:"
    for target_info in "${android_targets[@]}"; do
        IFS=':' read -r target name <<< "$target_info"
        echo "  target/$target/${BUILD_TYPE}/lib${PACKAGE//-/_}.so"
    done
}

# Function to build all iOS targets
build_ios() {
    print_info "=== Building for iOS ==="
    
    # Check if running on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "iOS builds require macOS"
        return 1
    fi
    
    local ios_targets=(
        "aarch64-apple-ios:iOS Device (ARM64)"
        "x86_64-apple-ios:iOS Simulator (Intel)"
        "aarch64-apple-ios-sim:iOS Simulator (Apple Silicon)"
    )
    
    for target_info in "${ios_targets[@]}"; do
        IFS=':' read -r target name <<< "$target_info"
        
        if ! check_target "$target"; then
            print_warn "Target $target not installed"
            install_target "$target"
        fi
        
        build_target "$target" "$name"
    done
    
    print_info "=== iOS build complete ==="
    print_info "Libraries located in:"
    for target_info in "${ios_targets[@]}"; do
        IFS=':' read -r target name <<< "$target_info"
        echo "  target/$target/${BUILD_TYPE}/lib${PACKAGE//-/_}.a"
    done
}

# Function to create universal iOS library
build_ios_universal() {
    print_info "=== Creating universal iOS libraries ==="
    
    # Check if running on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "iOS builds require macOS"
        return 1
    fi
    
    # Build all iOS targets first
    build_ios
    
    local lib_name="lib${PACKAGE//-/_}.a"
    local output_dir="target/universal/${BUILD_TYPE}"
    
    mkdir -p "$output_dir"
    
    # Create universal simulator library (Intel + Apple Silicon)
    print_info "Creating universal simulator library..."
    lipo -create \
        "target/x86_64-apple-ios/${BUILD_TYPE}/${lib_name}" \
        "target/aarch64-apple-ios-sim/${BUILD_TYPE}/${lib_name}" \
        -output "${output_dir}/${lib_name%.a}_sim.a"
    
    # Copy device library
    print_info "Copying device library..."
    cp "target/aarch64-apple-ios/${BUILD_TYPE}/${lib_name}" \
       "${output_dir}/${lib_name%.a}_device.a"
    
    print_info "=== Universal iOS libraries created ==="
    print_info "Libraries located in:"
    echo "  ${output_dir}/${lib_name%.a}_sim.a (Simulator)"
    echo "  ${output_dir}/${lib_name%.a}_device.a (Device)"
}

# Function to show usage
show_usage() {
    cat << EOF
TB Language - Mobile Build Script

Usage: $0 [OPTIONS] [PLATFORM]

PLATFORMS:
    android         Build for all Android targets
    ios             Build for all iOS targets
    ios-universal   Build universal iOS libraries
    all             Build for all platforms (Android + iOS)

OPTIONS:
    -d, --debug     Build in debug mode (default: release)
    -f, --features  Enable specific features (e.g., "mobile")
    -p, --package   Package to build (default: tb-runtime)
    -h, --help      Show this help message

EXAMPLES:
    $0 android                          # Build for Android (release)
    $0 ios                              # Build for iOS (release)
    $0 all                              # Build for all platforms
    $0 --debug android                  # Build for Android (debug)
    $0 --features mobile android        # Build with mobile features
    $0 --package tb-cli android         # Build tb-cli for Android

REQUIREMENTS:
    - Rust toolchain (rustup)
    - Android NDK (for Android builds)
    - Xcode (for iOS builds, macOS only)
    - Configured .cargo/config.toml with NDK paths

For more information, see MOBILE_BUILD.md
EOF
}

# Parse command line arguments
PLATFORM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="debug"
            shift
            ;;
        -f|--features)
            FEATURES="$2"
            shift 2
            ;;
        -p|--package)
            PACKAGE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        android|ios|ios-universal|all)
            PLATFORM="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_info "TB Language Mobile Build Script"
print_info "Package: $PACKAGE"
print_info "Build type: $BUILD_TYPE"
if [ -n "$FEATURES" ]; then
    print_info "Features: $FEATURES"
fi
echo ""

case $PLATFORM in
    android)
        build_android
        ;;
    ios)
        build_ios
        ;;
    ios-universal)
        build_ios_universal
        ;;
    all)
        build_android
        echo ""
        build_ios
        ;;
    "")
        print_error "No platform specified"
        show_usage
        exit 1
        ;;
    *)
        print_error "Unknown platform: $PLATFORM"
        show_usage
        exit 1
        ;;
esac

print_info ""
print_info "Build complete! 🚀"

