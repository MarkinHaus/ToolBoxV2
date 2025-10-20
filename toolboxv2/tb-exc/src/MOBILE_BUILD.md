# TB Language - Mobile Platform Compilation Guide

This guide explains how to compile the TB Language runtime and CLI for Android and iOS platforms.

## Overview

The TB Language project now supports cross-compilation to mobile platforms:
- **Android**: ARM64, ARMv7, x86, x86_64
- **iOS**: ARM64 (device), x86_64 (simulator), ARM64 (simulator)

## Prerequisites

### For Android

1. **Install Rust targets:**
   ```bash
   rustup target add aarch64-linux-android
   rustup target add armv7-linux-androideabi
   rustup target add i686-linux-android
   rustup target add x86_64-linux-android
   ```

2. **Install Android NDK:**
   - Download from: https://developer.android.com/ndk/downloads
   - Set environment variable: `ANDROID_NDK_HOME=/path/to/ndk`

3. **Configure NDK toolchain:**
   ```bash
   # Add to ~/.cargo/config.toml or .cargo/config.toml in project root
   [target.aarch64-linux-android]
   ar = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/llvm-ar"
   linker = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/aarch64-linux-android<API>-clang"
   
   [target.armv7-linux-androideabi]
   ar = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/llvm-ar"
   linker = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/armv7a-linux-androideabi<API>-clang"
   
   [target.i686-linux-android]
   ar = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/llvm-ar"
   linker = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/i686-linux-android<API>-clang"
   
   [target.x86_64-linux-android]
   ar = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/llvm-ar"
   linker = "<NDK_HOME>/toolchains/llvm/prebuilt/<HOST>/bin/x86_64-linux-android<API>-clang"
   ```
   
   Replace:
   - `<NDK_HOME>` with your NDK path
   - `<HOST>` with your platform (e.g., `linux-x86_64`, `darwin-x86_64`, `windows-x86_64`)
   - `<API>` with minimum API level (e.g., `21`, `24`, `28`)

### For iOS

1. **Install Rust targets:**
   ```bash
   rustup target add aarch64-apple-ios          # iOS devices
   rustup target add x86_64-apple-ios           # iOS simulator (Intel)
   rustup target add aarch64-apple-ios-sim      # iOS simulator (Apple Silicon)
   ```

2. **Install Xcode:**
   - Required for iOS development
   - Install from Mac App Store or developer.apple.com

3. **Install cargo-lipo (optional, for universal libraries):**
   ```bash
   cargo install cargo-lipo
   ```

## Building for Android

### Build tb-runtime as shared library

```bash
# ARM64 (most modern devices)
cargo build --release --target aarch64-linux-android -p tb-runtime

# ARMv7 (older devices)
cargo build --release --target armv7-linux-androideabi -p tb-runtime

# x86_64 (emulator)
cargo build --release --target x86_64-linux-android -p tb-runtime

# i686 (older emulator)
cargo build --release --target i686-linux-android -p tb-runtime
```

### Build all Android targets at once

```bash
for target in aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android; do
    cargo build --release --target $target -p tb-runtime
done
```

### Output locations

The compiled libraries will be in:
```
target/aarch64-linux-android/release/libtb_runtime.so
target/armv7-linux-androideabi/release/libtb_runtime.so
target/i686-linux-android/release/libtb_runtime.so
target/x86_64-linux-android/release/libtb_runtime.so
```

### Integration with Android project

1. Copy the `.so` files to your Android project:
   ```
   android/app/src/main/jniLibs/
   ├── arm64-v8a/
   │   └── libtb_runtime.so
   ├── armeabi-v7a/
   │   └── libtb_runtime.so
   ├── x86/
   │   └── libtb_runtime.so
   └── x86_64/
       └── libtb_runtime.so
   ```

2. Load the library in your Java/Kotlin code:
   ```java
   static {
       System.loadLibrary("tb_runtime");
   }
   ```

## Building for iOS

### Build tb-runtime as static library

```bash
# iOS device (ARM64)
cargo build --release --target aarch64-apple-ios -p tb-runtime

# iOS simulator (Intel)
cargo build --release --target x86_64-apple-ios -p tb-runtime

# iOS simulator (Apple Silicon)
cargo build --release --target aarch64-apple-ios-sim -p tb-runtime
```

### Build universal library (all iOS targets)

Using cargo-lipo:
```bash
cargo lipo --release -p tb-runtime
```

Or manually create a universal library:
```bash
# Build for all targets
cargo build --release --target aarch64-apple-ios -p tb-runtime
cargo build --release --target x86_64-apple-ios -p tb-runtime
cargo build --release --target aarch64-apple-ios-sim -p tb-runtime

# Create universal library for simulators
lipo -create \
    target/x86_64-apple-ios/release/libtb_runtime.a \
    target/aarch64-apple-ios-sim/release/libtb_runtime.a \
    -output target/universal/release/libtb_runtime_sim.a

# Device library
cp target/aarch64-apple-ios/release/libtb_runtime.a \
   target/universal/release/libtb_runtime_device.a
```

### Output locations

The compiled libraries will be in:
```
target/aarch64-apple-ios/release/libtb_runtime.a
target/x86_64-apple-ios/release/libtb_runtime.a
target/aarch64-apple-ios-sim/release/libtb_runtime.a
```

### Integration with iOS project

1. Add the static library to your Xcode project
2. Link against the library in Build Phases
3. Add header files for FFI bindings

## Library Types

The mobile builds produce the following library types:

- **cdylib**: Dynamic library (`.so` for Android, `.dylib` for iOS)
- **staticlib**: Static library (`.a` for both platforms)
- **rlib**: Rust library (for Rust-to-Rust linking)

## Features

### Mobile-specific features

Enable mobile features when building:
```bash
cargo build --release --target aarch64-linux-android -p tb-runtime --features mobile
```

### Disable desktop-only features

Some features like the REPL are automatically disabled on mobile platforms through conditional compilation.

## Troubleshooting

### Android NDK not found
```
error: linker `aarch64-linux-android-clang` not found
```
**Solution:** Set `ANDROID_NDK_HOME` environment variable and configure `.cargo/config.toml`

### iOS signing issues
```
error: failed to sign the binary
```
**Solution:** Configure code signing in Xcode or use `--no-sign` flag

### Missing dependencies
```
error: package `jni` cannot be built because it requires rustc 1.XX or newer
```
**Solution:** Update Rust: `rustup update`

### Cross-compilation errors
**Solution:** Ensure you have the correct target installed: `rustup target list --installed`

## Performance Considerations

- **Release builds**: Always use `--release` for mobile deployments
- **LTO**: Link-Time Optimization is enabled by default in release profile
- **Strip**: Symbols are stripped to reduce binary size
- **Optimization level**: Set to 3 for maximum performance

## Size Optimization

To reduce library size:

1. **Enable strip in Cargo.toml** (already configured):
   ```toml
   [profile.release]
   strip = true
   ```

2. **Use cargo-bloat to analyze size**:
   ```bash
   cargo install cargo-bloat
   cargo bloat --release --target aarch64-linux-android -p tb-runtime
   ```

3. **Consider feature flags** to exclude unused functionality

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Mobile Build

on: [push, pull_request]

jobs:
  android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Install Android targets
        run: |
          rustup target add aarch64-linux-android
          rustup target add armv7-linux-androideabi
      - name: Build Android
        run: |
          cargo build --release --target aarch64-linux-android -p tb-runtime
          cargo build --release --target armv7-linux-androideabi -p tb-runtime

  ios:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Install iOS targets
        run: |
          rustup target add aarch64-apple-ios
          rustup target add x86_64-apple-ios
      - name: Build iOS
        run: |
          cargo build --release --target aarch64-apple-ios -p tb-runtime
          cargo build --release --target x86_64-apple-ios -p tb-runtime
```

## Next Steps

- Create FFI bindings for your mobile app
- Implement JNI wrappers for Android
- Create Objective-C/Swift bindings for iOS
- Test on physical devices
- Profile performance and optimize

## Resources

- [Rust Cross-Compilation Guide](https://rust-lang.github.io/rustup/cross-compilation.html)
- [Android NDK Documentation](https://developer.android.com/ndk/guides)
- [iOS Development with Rust](https://mozilla.github.io/firefox-browser-architecture/experiments/2017-09-06-rust-on-ios.html)
- [cargo-ndk](https://github.com/bbqsrc/cargo-ndk) - Simplified Android builds
- [cargo-lipo](https://github.com/TimNN/cargo-lipo) - iOS universal libraries

---

**TB Language v0.1.0** - *Fast, Safe, Interoperable* 🚀

*Now available on Android and iOS!*

