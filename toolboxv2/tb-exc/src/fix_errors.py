import re

files_to_fix = [
    ('crates/tb-plugin/src/loader.rs', 'PluginError', 'plugin_error'),
    ('crates/tb-plugin/src/compiler.rs', 'PluginError', 'plugin_error'),
    ('crates/tb-plugin/src/compiler.rs', 'CompilationError', 'compilation_error'),
    ('crates/tb-builtins/src/lib.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-builtins/src/blob.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-builtins/src/utils.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-builtins/src/file_io.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-builtins/src/networking.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-codegen/src/rust_codegen.rs', 'PluginError', 'plugin_error'),
    ('crates/tb-codegen/src/rust_codegen.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-types/src/checker.rs', 'TypeError', 'type_error'),
    ('crates/tb-jit/src/executor.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-jit/src/executor.rs', 'PluginError', 'plugin_error'),
    ('crates/tb-jit/src/builtins.rs', 'RuntimeError', 'runtime_error'),
    ('crates/tb-cli/src/runner.rs', 'RuntimeError', 'runtime_error'),
]

def find_matching_brace(text, start_pos):
    """Find the matching closing brace for an opening brace at start_pos"""
    count = 1
    pos = start_pos + 1
    while pos < len(text) and count > 0:
        if text[pos] == '{':
            count += 1
        elif text[pos] == '}':
            count -= 1
        pos += 1
    return pos - 1 if count == 0 else -1

def fix_error_patterns(content, error_type, helper_func):
    """Fix all error patterns in the content using manual parsing"""

    # Find all occurrences of TBError::ErrorType {
    pattern = rf'TBError::{error_type}\s*\{{'

    result = []
    last_end = 0

    for match in re.finditer(pattern, content):
        start = match.start()
        brace_start = match.end() - 1  # Position of '{'

        # Find the matching closing brace
        brace_end = find_matching_brace(content, brace_start)
        if brace_end == -1:
            continue

        # Extract the content between braces
        inner_content = content[brace_start+1:brace_end]

        # Check if this is a message field
        message_match = re.search(r'message:\s*(.+?),?\s*$', inner_content, re.DOTALL)
        if not message_match:
            continue

        message_value = message_match.group(1).strip()

        # Remove trailing comma if present
        if message_value.endswith(','):
            message_value = message_value[:-1].strip()

        # Create the replacement
        replacement = f'TBError::{helper_func}({message_value})'

        # Add the content before this match
        result.append(content[last_end:start])
        # Add the replacement
        result.append(replacement)
        # Update last_end
        last_end = brace_end + 1

    # Add the remaining content
    result.append(content[last_end:])

    return ''.join(result)

for file_path, error_type, helper_func in files_to_fix:
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix all patterns
    content = fix_error_patterns(content, error_type, helper_func)

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'Fixed all TBError::{error_type} in {file_path}')

