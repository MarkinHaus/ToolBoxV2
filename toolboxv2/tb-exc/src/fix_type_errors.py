#!/usr/bin/env python3
"""
Fix TBError::type_error calls in checker.rs to use self.type_error_with_context with span
"""

import re
import os

def fix_type_errors_in_checker():
    file_path = "crates/tb-types/src/checker.rs"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Manual replacements for specific patterns
    replacements = [
        # For loop - Statement::For has span
        (
            r'Statement::For \{ variable, iterable, body, \.\. \}',
            r'Statement::For { variable, iterable, body, span }'
        ),
        (
            r'_ => return Err\(TBError::type_error\(format!\(\s*"Cannot iterate over \{:?\?\}", iter_type\s*\)\)\)',
            r'_ => return Err(self.type_error_with_context(format!("Cannot iterate over {:?}", iter_type), Some(*span)))'
        ),
        
        # While loop - Statement::While has span
        (
            r'Statement::While \{ condition, body, \.\. \}',
            r'Statement::While { condition, body, span }'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"While condition must be bool, got \{:?\?\}", cond_type\s*\)\)\);',
            r'return Err(self.type_error_with_context(format!("While condition must be bool, got {:?}", cond_type), Some(*span)));'
        ),
        
        # Expression::Call has span
        (
            r'Expression::Call \{ callee, args, \.\. \}',
            r'Expression::Call { callee, args, span }'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"Expected \{\} arguments, got \{\}",\s*params\.len\(\),',
            r'return Err(self.type_error_with_context(format!("Expected {} arguments, got {}", params.len(),'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"Argument type mismatch: expected \{:?\?\}, got \{:?\?\}",\s*param_type, arg_type',
            r'return Err(self.type_error_with_context(format!("Argument type mismatch: expected {:?}, got {:?}", param_type, arg_type'
        ),
        (
            r'_ => Err\(TBError::type_error\(format!\(\s*"Cannot call non-function type \{:?\?\}", func_type\s*\)\)\)',
            r'_ => Err(self.type_error_with_context(format!("Cannot call non-function type {:?}", func_type), Some(*span)))'
        ),
        
        # Expression::List has span
        (
            r'Expression::List \{ elements, \.\. \}',
            r'Expression::List { elements, span }'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"List elements must have compatible types, got \{:?\?\} and \{:?\?\}",\s*first_type, elem_type',
            r'return Err(self.type_error_with_context(format!("List elements must have compatible types, got {:?} and {:?}", first_type, elem_type'
        ),
        
        # Expression::Index has span
        (
            r'Expression::Index \{ object, index, \.\. \}',
            r'Expression::Index { object, index, span }'
        ),
        (
            r'return Err\(TBError::type_error\("List index must be int"\.to_string\(\)\)\);',
            r'return Err(self.type_error_with_context("List index must be int".to_string(), Some(*span)));'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"Dict key type mismatch: expected \{:?\?\}, got \{:?\?\}",\s*key_type, idx_type',
            r'return Err(self.type_error_with_context(format!("Dict key type mismatch: expected {:?}, got {:?}", key_type, idx_type'
        ),
        (
            r'Err\(TBError::type_error\(format!\("Cannot index type Generic\(\{\}\)", name\)\)\)',
            r'Err(self.type_error_with_context(format!("Cannot index type Generic({})", name), Some(*span)))'
        ),
        (
            r'ref other => Err\(TBError::type_error\(format!\("Cannot index type \{:?\?\}", other\)\)\)',
            r'ref other => Err(self.type_error_with_context(format!("Cannot index type {:?}", other), Some(*span)))'
        ),
        
        # Expression::Match has span
        (
            r'Expression::Match \{ value, arms, \.\. \}',
            r'Expression::Match { value, arms, span }'
        ),
        (
            r'return Err\(TBError::type_error\(\s*"Match expression must have at least one arm"\.to_string\(\)\s*\)\);',
            r'return Err(self.type_error_with_context("Match expression must have at least one arm".to_string(), Some(*span)));'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"Match arms have incompatible types: \{:?\?\} vs \{:?\?\}",\s*first_arm_type, arm_type',
            r'return Err(self.type_error_with_context(format!("Match arms have incompatible types: {:?} vs {:?}", first_arm_type, arm_type'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"Pattern type \{:?\?\} doesn\'t match value type \{:?\?\}",\s*pattern_type, value_type',
            r'return Err(self.type_error_with_context(format!("Pattern type {:?} doesn\'t match value type {:?}", pattern_type, value_type'
        ),
        (
            r'return Err\(TBError::type_error\(format!\(\s*"Range pattern requires Int type, got \{:?\?\}",\s*value_type',
            r'return Err(self.type_error_with_context(format!("Range pattern requires Int type, got {:?}", value_type'
        ),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Fix the closing parentheses for multi-line error calls
    # Add ), Some(*span)) at the end of incomplete calls
    content = re.sub(
        r'(self\.type_error_with_context\(format!\("Expected \{\} arguments[^)]+\)\)),\s*args\.len\(\)\s*\)\);',
        r'\1, args.len())), Some(*span));',
        content
    )
    content = re.sub(
        r'(self\.type_error_with_context\(format!\("Argument type mismatch[^)]+\)\)),\s*\)\);',
        r'\1)), Some(*span));',
        content
    )
    content = re.sub(
        r'(self\.type_error_with_context\(format!\("List elements must have compatible types[^)]+\)\)),\s*\)\);',
        r'\1)), Some(*span));',
        content
    )
    content = re.sub(
        r'(self\.type_error_with_context\(format!\("Dict key type mismatch[^)]+\)\)),\s*\)\);',
        r'\1)), Some(*span));',
        content
    )
    content = re.sub(
        r'(self\.type_error_with_context\(format!\("Match arms have incompatible types[^)]+\)\)),\s*\)\);',
        r'\1)), Some(*span));',
        content
    )
    content = re.sub(
        r'(self\.type_error_with_context\(format!\("Pattern type[^)]+\)\)),\s*\)\);',
        r'\1)), Some(*span));',
        content
    )
    content = re.sub(
        r'(self\.type_error_with_context\(format!\("Range pattern requires[^)]+\)\)),\s*\)\);',
        r'\1)), Some(*span));',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed type errors in {file_path}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    fix_type_errors_in_checker()

