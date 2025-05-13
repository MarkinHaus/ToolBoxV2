import ast
import os
import re

import astor  # More robust AST handling


class CodeProcessor:
    def __init__(self, code_base='./'):
        self.language_patterns = [
            r'```([\w-]+)\n((?:#|//|<!--)\s*(\S+))?\n([\s\S]*?)```',  # Standard pattern
            r'```([\w-]+)\s*\n([\s\S]*?)```'  # Pattern without filename comment
        ]
        self.code_base = code_base

    def extract_code(self, text):
        code_blocks = {}
        seen = set()
        for pattern in self.language_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:

                print(match.groups())

                if len(match.groups()) < 3:
                    continue

                code = match.groups()[3]
                filename = match.groups()[2]

                if code == code_blocks.get(filename):
                    continue

                if code_blocks.get(filename) is not None and code != code_blocks.get(filename):
                    comment_prfix = match.groups()[1].replace(filename, '')
                    filename = code.split('\n')[0].replace(comment_prfix, '')
                    code = code.replace(comment_prfix + filename + '\n', '')

                    print("new code", code)

                seen.add(filename)

                code_blocks[filename] = code
        return code_blocks

    def write_code(self, code_dict):
        for filename, code in code_dict.items():
            filepath = os.path.join(self.code_base, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            print("Writing", filepath)
            with open(filepath, "w") as f:
                f.write(code)

    def extract_and_write_code(self, text):
        code_blocks = self.extract_code(text)
        files = []
        for filename, new_code in code_blocks.items():
            filepath = os.path.join(self.code_base, filename)
            files.append(filepath)
            if os.path.exists(filepath):
                self.update_existing_file(filepath, new_code)
            else:
                self.write_code({filename: new_code})
        return files

    def update_existing_file(self, filepath, new_code):
        """
            Update an existing Python file with new code while preserving existing implementations.

            Args:
                filepath (str): Path to the file to be updated
                new_code (str): New code to merge with existing code
            """
        try:
            # Read existing code
            with open(filepath) as f:
                existing_code = f.read()

            # Parse existing and new code
            existing_ast_tree = ast.parse(existing_code)
            new_ast_tree = ast.parse(new_code)

            # Create updater and transform the AST
            updater = CodeUpdater(existing_ast_tree)
            updated_ast = updater.visit(new_ast_tree)

            # Convert AST back to source code
            updated_code = astor.to_source(updated_ast)

            # Write updated code back to file
            with open(filepath, 'w') as f:
                f.write(updated_code)

            print(f"Successfully updated {filepath}")
            return True

        except Exception as e:
            print(f"Error updating {filepath}: {e}")
            return False


class CodeUpdater(ast.NodeTransformer):
    def __init__(self, existing_ast):
        self.existing_ast = existing_ast
        self.updated_nodes = set()

    def visit_Module(self, node):
        # Process module-level nodes
        new_body = []
        for new_node in node.body:
            updated_node = self.visit(new_node)
            if updated_node is not None:
                new_body.append(updated_node)
        node.body = new_body
        return node

    def visit_ClassDef(self, node):
        # Find existing class with the same name
        existing_class = next(
            (c for c in self.existing_ast.body
             if isinstance(c, ast.ClassDef) and c.name == node.name),
            None
        )

        if existing_class:
            # Merge existing and new class attributes and methods
            merged_body = []
            existing_items = {
                type(item).__name__ + '_' + getattr(item, 'name', ''): item
                for item in existing_class.body
            }

            for new_item in node.body:
                key = type(new_item).__name__ + '_' + getattr(new_item, 'name', '')

                # If item already exists, keep the existing implementation
                if key in existing_items and isinstance(new_item, ast.FunctionDef | ast.AsyncFunctionDef):
                    merged_body.append(existing_items[key])
                    self.updated_nodes.add(key)
                else:
                    # Add new items not in existing class
                    merged_body.append(new_item)

            # Add any existing items that weren't updated
            for key, item in existing_items.items():
                if key not in self.updated_nodes:
                    merged_body.append(item)

            # Update class decorators, bases, and keywords
            node.decorator_list = node.decorator_list or existing_class.decorator_list
            node.bases = node.bases or existing_class.bases
            node.keywords = node.keywords or existing_class.keywords

            node.body = merged_body

        return node

    def visit_FunctionDef(self, node):
        # Find existing function with the same name at module or class level
        existing_func = next(
            (f for f in self.existing_ast.body
             if isinstance(f, ast.FunctionDef) and f.name == node.name),
            None
        )

        if existing_func:
            # Preserve existing function implementation if not explicitly changed
            node.body = existing_func.body
            node.decorator_list = node.decorator_list or existing_func.decorator_list

        return node


code_processors = {}


def extract_code_blocks(markdown_string, base_dir):
    cp = code_processors.get(base_dir, CodeProcessor(base_dir))
    return cp.extract_and_write_code(markdown_string)
