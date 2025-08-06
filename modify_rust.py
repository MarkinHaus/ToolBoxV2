#!/usr/bin/env python3
import re

# Read the main.rs file
with open('/home/daytona/ToolBoxV2/toolboxv2/src-core/src/main.rs', 'r') as f:
    content = f.read()

# Add websocket_enabled field to ServerSettings struct
pattern = r'(    watch_modules: Vec<String>,)\n(})'
replacement = r'\1\n    websocket_enabled: bool,\n\2'
content = re.sub(pattern, replacement, content)

# Write the modified content back
with open('/home/daytona/ToolBoxV2/toolboxv2/src-core/src/main.rs', 'w') as f:
    f.write(content)

print('Successfully added websocket_enabled field to ServerSettings struct')
