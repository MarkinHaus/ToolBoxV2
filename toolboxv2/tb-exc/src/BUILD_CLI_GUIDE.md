# TB Language - CLI Build Guide

This guide explains how to use the enhanced `tb run build` command to compile TB Language for various platforms.

## Overview

The TB Language CLI now supports building for multiple platforms with a single command:
- **Desktop**: Windows, Linux, macOS (Intel & Apple Silicon)
- **Mobile**: Android (all architectures), iOS (device & simulator)
- **Cross-compilation**: Build for any platform from any host

## Quick Start

### Build for Current Platform (Native)

```bash
# Release build (optimized)
tb run build

# Debug build (with symbols)
tb run build --debug
```

### Build for Specific Platform

```bash
# Desktop platforms
tb run build --target windows      # Windows x64
tb run build --target linux        # Linux x64
tb run build --target macos        # macOS Intel
tb run build --target macos-arm    # macOS Apple Silicon

# Mobile platforms
tb run build --target android      # All Android architectures
tb run build --target ios          # All iOS targets (macOS only)

# Build everything
tb run build --target all          # All supported platforms
```

## Command Options

### `--target` (Platform Selection)

Specifies which platform to build for:

| Target | Description | Output |
|--------|-------------|--------|
| `native` | Current platform (default) | Single executable |
| `windows` | Windows x64 | `tb.exe` |
| `linux` | Linux x64 | `tb` |
| `macos` | macOS Intel | `tb` |
| `macos-arm` | macOS Apple Silicon | `tb` |
| `android` | All Android architectures | 4 `.so` libraries |
| `ios` | All iOS targets | 3 `.a` libraries |
| `all` | All platforms | All of the above |

### `--debug` (Build Mode)

Build in debug mode instead of release:

```bash
tb run build --debug
tb run build --target android --debug
```

**Differences:**
- **Release**: Optimized, stripped, smaller size, slower compilation
- **Debug**: Symbols included, faster compilation, larger size, easier debugging

### `--no-export` (Skip Export)

Skip automatic export to `bin/` directory:

```bash
tb run build --no-export
```

By default, all builds are automatically exported to the `bin/` directory.

## Output Locations

### Native Builds

Executables are placed in:
```
bin/
└── tb(.exe)              # Native executable
```

### Cross-Platform Desktop Builds

```
bin/
├── x86_64-pc-windows-msvc/
│   └── tb.exe
├── x86_64-unknown-linux-gnu/
│   └── tb
├── x86_64-apple-darwin/
│   └── tb
└── aarch64-apple-darwin/
    └── tb
```

### Android Builds

```
bin/android/
├── arm64-v8a/
│   └── libtb_runtime.so
├── armeabi-v7a/
│   └── libtb_runtime.so
├── x86/
│   └── libtb_runtime.so
└── x86_64/
    └── libtb_runtime.so
```

### iOS Builds

```
bin/ios/
├── device/
│   └── libtb_runtime.a
├── simulator-intel/
│   └── libtb_runtime.a
└── simulator-arm64/
    └── libtb_runtime.a
```

## Platform-Specific Requirements

### Windows

**For native builds:**
- Rust toolchain with MSVC target

**For cross-compilation to Windows:**
```bash
rustup target add x86_64-pc-windows-msvc
```

### Linux

**For native builds:**
- Rust toolchain with GNU target

**For cross-compilation to Linux:**
```bash
rustup target add x86_64-unknown-linux-gnu
```

### macOS

**For native builds:**
- Rust toolchain with Darwin target
- Xcode Command Line Tools

**For cross-compilation to macOS:**
```bash
rustup target add x86_64-apple-darwin
rustup target add aarch64-apple-darwin
```

### Android

**Requirements:**
1. Android NDK installed
2. `ANDROID_NDK_HOME` environment variable set
3. `.cargo/config.toml` configured (see `.cargo/config.toml.example`)

**Install targets:**
```bash
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add i686-linux-android
rustup target add x86_64-linux-android
```

**Build:**
```bash
tb run build --target android
```

The CLI automatically uses `build-mobile.sh` (Linux/macOS) or `build-mobile.ps1` (Windows).

### iOS

**Requirements:**
1. macOS with Xcode installed
2. iOS SDK

**Install targets:**
```bash
rustup target add aarch64-apple-ios
rustup target add x86_64-apple-ios
rustup target add aarch64-apple-ios-sim
```

**Build:**
```bash
tb run build --target ios
```

**Note:** iOS builds only work on macOS.

## Examples

### Example 1: Build for Current Platform

```bash
# Simple release build
tb run build

# Output:
# ✓ Build successful!
# ✓ Exported to: /path/to/toolboxv2/bin/tb
```

### Example 2: Build for Android

```bash
# Build all Android architectures
tb run build --target android

# Output:
# ✓ Build successful for Android!
# ✓ Exported arm64-v8a: /path/to/bin/android/arm64-v8a/libtb_runtime.so
# ✓ Exported armeabi-v7a: /path/to/bin/android/armeabi-v7a/libtb_runtime.so
# ✓ Exported x86: /path/to/bin/android/x86/libtb_runtime.so
# ✓ Exported x86_64: /path/to/bin/android/x86_64/libtb_runtime.so
```

### Example 3: Build for All Platforms

```bash
# Build everything (takes time!)
tb run build --target all

# Builds:
# - Native executable
# - Android libraries (4 architectures)
# - iOS libraries (3 targets, macOS only)
```

### Example 4: Debug Build for Testing

```bash
# Debug build with symbols
tb run build --debug

# Faster compilation, larger binary, includes debug symbols
```

### Example 5: Cross-Compile for Windows from Linux

```bash
# Install Windows target
rustup target add x86_64-pc-windows-msvc

# Build for Windows
tb run build --target windows

# Output: bin/x86_64-pc-windows-msvc/tb.exe
```

## Integration with Mobile Apps

### Android Integration

After building for Android:

1. Copy libraries to your Android project:
   ```bash
   cp -r bin/android/* android-app/src/main/jniLibs/
   ```

2. Load in Java/Kotlin:
   ```java
   static {
       System.loadLibrary("tb_runtime");
   }
   ```

### iOS Integration

After building for iOS:

1. Add libraries to Xcode project
2. Link against the appropriate library (device or simulator)
3. Create Swift/Objective-C bindings

## Troubleshooting

### Build Script Not Found

**Error:**
```
Mobile build script not found: /path/to/build-mobile.sh
```

**Solution:**
Ensure `build-mobile.sh` and `build-mobile.ps1` are in the project directory:
```bash
ls tb-exc/src/build-mobile.*
```

### Target Not Installed

**Error:**
```
error: target 'aarch64-linux-android' not found
```

**Solution:**
Install the required target:
```bash
rustup target add aarch64-linux-android
```

### Android NDK Not Configured

**Error:**
```
error: linker 'aarch64-linux-android-clang' not found
```

**Solution:**
1. Install Android NDK
2. Set `ANDROID_NDK_HOME` environment variable
3. Configure `.cargo/config.toml` (see `.cargo/config.toml.example`)

### iOS Build on Non-macOS

**Error:**
```
iOS builds require macOS
```

**Solution:**
iOS builds can only be performed on macOS with Xcode installed.

## Advanced Usage

### Custom Build Scripts

The CLI uses these scripts for mobile builds:
- **Linux/macOS**: `build-mobile.sh`
- **Windows**: `build-mobile.ps1`

You can modify these scripts for custom build configurations.

### Manual Builds

You can also build manually using cargo:

```bash
# Native
cargo build --release

# Specific target
cargo build --release --target aarch64-linux-android

# Using mobile script
./build-mobile.sh android
```

## Performance Tips

1. **Use release builds for production**: `tb run build` (default)
2. **Use debug builds for development**: `tb run build --debug`
3. **Build only what you need**: Avoid `--target all` unless necessary
4. **Enable LTO**: Already enabled in release profile
5. **Strip symbols**: Already enabled in release profile

## See Also

- [MOBILE_BUILD.md](MOBILE_BUILD.md) - Detailed mobile compilation guide
- [README.md](README.md) - Main project documentation
- `.cargo/config.toml.example` - NDK configuration template

---

**TB Language v0.1.0** - *Fast, Safe, Interoperable* 🚀

*Build once, run everywhere!*

