# TB Language Setup Tests

Test suite for TB Language setup scripts and IDE integrations.

## Test Files

### test_setup.py
Comprehensive test suite for setup.py and install_support.py:

- **TestTBxSetup**: Tests for TBxSetup class (file associations)
  - Initialization and path detection
  - Executable path detection
  - Icon path detection
  
- **TestTBSetup**: Tests for TBSetup class (complete installation)
  - Path validation
  - VS Code extension path
  - PyCharm plugin path

- **TestVSCodeExtension**: Tests for VS Code extension configuration
  - package.json validation
  - language-configuration.json validation
  - Syntax file (tb.tmLanguage.json) validation
  - File extension configuration (.tbx and .tb)

- **TestPyCharmPlugin**: Tests for PyCharm plugin configuration
  - plugin.xml validation
  - TB.xml file type validation
  - Comment syntax validation (// and /* */)
  - File extension configuration (.tbx and .tb)

### validate_documentation.py
Documentation consistency validator:

- File extension documentation (.tbx and .tb)
- Comment syntax documentation (// and /* */)
- Version consistency across all files
- Execution mode documentation (JIT and AOT)
- Keyword consistency across implementations

## Running Tests

### Run all tests
```bash
python -m pytest toolboxv2/utils/tbx/test/test_setup.py -v
```

### Run specific test class
```bash
python -m pytest toolboxv2/utils/tbx/test/test_setup.py::TestVSCodeExtension -v
```

### Run documentation validation
```bash
python toolboxv2/utils/tbx/test/validate_documentation.py
```

## Test Results

**Last Run: 2025-11-10**

- ✅ **16/16 tests passed** in test_setup.py
- ✅ **Documentation validation passed** with 2 warnings

### Known Warnings
1. Lang.md may reference # comments (legacy documentation)
2. Lang.md doesn't explicitly mention AOT mode (uses "compiled" instead)

## What Was Fixed

### Critical Issues Fixed
1. ✅ **Empty syntax file**: tb.tmLanguage.json was completely empty - now has full TextMate grammar
2. ✅ **Wrong comment syntax**: Both VS Code and PyCharm used `#` instead of `//` and `/* */`
3. ✅ **Missing .tb extension**: Only .tbx was supported, now both .tbx and .tb work
4. ✅ **Incorrect paths**: install_support.py referenced non-existent paths

### Files Updated
- `toolboxv2/utils/tbx/setup.py` - v1.0.1
- `toolboxv2/utils/tbx/install_support.py` - v1.0.1
- `toolboxv2/utils/tbx/tb-lang-support/syntaxes/tb.tmLanguage.json` - Complete rewrite
- `toolboxv2/utils/tbx/tb-lang-support/package.json` - v1.0.1
- `toolboxv2/utils/tbx/tb-lang-support/language-configuration.json` - Fixed comment syntax
- `toolboxv2/utils/tbx/tb-lang-pycharm/src/main/resources/fileTypes/TB.xml` - Fixed comment syntax
- `toolboxv2/utils/tbx/tb-lang-pycharm/src/main/resources/META-INF/plugin.xml` - v1.0.1

## Next Steps

To complete the setup:

1. **Build the TB compiler**:
   ```bash
   cd toolboxv2/tb-exc
   cargo build --release
   ```

2. **Run the complete setup**:
   ```bash
   python toolboxv2/utils/tbx/install_support.py
   ```

3. **Test VS Code extension**:
   ```bash
   cd toolboxv2/utils/tbx/tb-lang-support
   npm install
   npm run compile
   code --install-extension .
   ```

4. **Test PyCharm plugin**:
   ```bash
   cd toolboxv2/utils/tbx/tb-lang-pycharm
   python build_plugin.py
   python install_filetype.py
   ```

## Continuous Integration

These tests should be run:
- Before committing changes to setup scripts
- After updating TB Language syntax
- When adding new keywords or features
- Before releasing new versions

## Contributing

When adding new features to TB Language:

1. Update the syntax files (tb.tmLanguage.json, TB.xml)
2. Update the tests to verify the new features
3. Run all tests to ensure nothing broke
4. Update documentation
5. Run documentation validation

