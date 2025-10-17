# TB Language - PyCharm Plugin

## Installation

### Automatic
```bash
python3 setup_tb.py
```

### Manual

1. Build plugin:
   ```bash
   cd tb-lang-pycharm
   python3 build_plugin.py
   ```

2. Install in PyCharm:
   - Open PyCharm
   - File → Settings → Plugins
   - Click gear icon → Install Plugin from Disk
   - Select `tb-language.jar`
   - Restart PyCharm

## Features

- ✅ Syntax highlighting for .tbx files
- ✅ File type recognition
- ✅ Code completion
- ✅ Run configurations
- ✅ Context menu actions

## Usage

1. Open any `.tbx` file
2. Right-click → Run 'filename.tbx'
3. Or use keyboard shortcut: Ctrl+Shift+R

## Development

To modify the plugin:

1. Edit files in `src/main/resources/`
2. Run `python3 build_plugin.py`
3. Reinstall JAR in PyCharm

### Alternative

### Import in PyCharm: pycharm_external_tool.xml

Settings → Tools → External Tools
Click gear icon → Import
Select pycharm_external_tool.xml
Now you can: Right-click .tbx file → External Tools → Run TB Program
